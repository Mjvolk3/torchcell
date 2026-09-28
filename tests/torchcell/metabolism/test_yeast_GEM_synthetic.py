# tests/torchcell/metabolism/test_yeast_GEM_synthetic.py
# [[tests.torchcell.metabolism.test_yeast_GEM_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/metabolism/test_yeast_GEM_synthetic.py
"""YeastGEM over a three-reaction cobra model written to SBML under tmp_path.

The toy (built in :func:`_toy_model`, written with ``cobra.io.write_sbml_model`` to
``<root>/yeast-GEM-9.0.2/model/yeast-GEM.xml`` so ``YeastGEM(root=tmp)`` finds the
checkout and never downloads):

* ``R1: A_c <=> B_c`` (reversible), GPR ``(g1 and g2) or g3``;
* ``R2: 2 B_c --> C_c`` (irreversible), GPR ``g4``;
* ``EX_A: A_c <=>`` (reversible exchange), no GPR.

So the hypergraph and the bipartite graph have R1 as two gene combinations
(``{g1, g2}``, ``{g3}``) times two directions, R2 one forward edge, and EX_A a forward and
a reverse ``noGene`` edge: 7 reaction units. Every bipartite edge points reaction ->
metabolite; ``edge_type`` is ``reactant`` for a negative coefficient on the forward node
and flips on the reverse node; ``stoichiometry`` is the absolute coefficient (R2's
``B_c`` edge carries 2.0). The download test stubs ``requests.get`` at its import site
(``torchcell.metabolism.yeast_GEM.requests``) with an in-memory zip of the same SBML.
No solver is invoked anywhere in this file.
"""

import io
import os
import random
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cobra
import hypernetx as hnx
import pytest
from networkx.drawing.layout import spectral_layout

import torchcell.metabolism.yeast_GEM as yeast_gem_module
from torchcell.metabolism.yeast_GEM import (
    YeastGEM,
    analyze_reactions_without_genes,
    plot_bipartite_network,
    plot_full_network,
    plot_random_network,
    plot_reaction_map,
    sanity_check_metabolic_networks,
)
from torchcell.sequence import GeneSet


def _toy_model() -> cobra.Model:
    model = cobra.Model("toy")
    a = cobra.Metabolite(
        "A_c", name="alpha", formula="C6H12O6", compartment="c", charge=0
    )
    b = cobra.Metabolite(
        "B_c", name="beta", formula="C3H4O3", compartment="c", charge=-1
    )
    c = cobra.Metabolite(
        "C_c", name="gamma", formula="C6H8O6", compartment="c", charge=-2
    )
    r1 = cobra.Reaction("R1", lower_bound=-1000.0, upper_bound=1000.0)
    r1.add_metabolites({a: -1.0, b: 1.0})
    r2 = cobra.Reaction("R2", lower_bound=0.0, upper_bound=1000.0)
    r2.add_metabolites({b: -2.0, c: 1.0})
    ex = cobra.Reaction("EX_A", lower_bound=-10.0, upper_bound=1000.0)
    ex.add_metabolites({a: -1.0})
    model.add_reactions([r1, r2, ex])
    r1.gene_reaction_rule = "(g1 and g2) or g3"
    r2.gene_reaction_rule = "g4"
    return model


def _write_checkout(root: Path) -> Path:
    model_dir = root / "yeast-GEM-9.0.2" / "model"
    model_dir.mkdir(parents=True)
    path = model_dir / "yeast-GEM.xml"
    cobra.io.write_sbml_model(_toy_model(), str(path))
    return path


@pytest.fixture
def gem(tmp_path: Path) -> YeastGEM:
    """YeastGEM over the toy checkout (no download: model_dir exists)."""
    _write_checkout(tmp_path)
    return YeastGEM(root=str(tmp_path))


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

R1_FWD = {
    "reaction_id": "R1",
    "direction": "forward",
    "reactants": ["A_c"],
    "products": ["B_c"],
    "equation": "A_c <=> B_c",
    "reversibility": True,
    "stoichiometry": [-1.0, 1.0],
    "stoich_coefficient-A_c": -1.0,
    "stoich_coefficient-B_c": 1.0,
    "weight": 1,
}
R1_REV = {
    "reaction_id": "R1",
    "direction": "reverse",
    "reactants": ["B_c"],
    "products": ["A_c"],
    "equation": "A_c <=> B_c",
    "reversibility": True,
    "stoichiometry": [1.0, -1.0],
    "stoich_coefficient-A_c": 1.0,
    "stoich_coefficient-B_c": -1.0,
    "weight": 1,
}
R2_FWD = {
    "genes": {"g4"},
    "reaction_id": "R2",
    "direction": "forward",
    "reactants": ["B_c"],
    "products": ["C_c"],
    "equation": "2.0 B_c --> C_c",
    "reversibility": False,
    "stoichiometry": [-2.0, 1.0],
    "stoich_coefficient-B_c": -2.0,
    "stoich_coefficient-C_c": 1.0,
    "weight": 1,
}
EX_FWD = {
    "genes": set(),
    "reaction_id": "EX_A",
    "direction": "forward",
    "reactants": ["A_c"],
    "products": [],
    "equation": "A_c <=> ",
    "reversibility": True,
    "stoichiometry": [-1.0],
    "stoich_coefficient-A_c": -1.0,
    "weight": 1,
}
EX_REV = {
    "genes": set(),
    "reaction_id": "EX_A",
    "direction": "reverse",
    "reactants": [],
    "products": ["A_c"],
    "equation": "A_c <=> ",
    "reversibility": True,
    "stoichiometry": [1.0],
    "stoich_coefficient-A_c": 1.0,
    "weight": 1,
}


def _hyperedges(gem: YeastGEM) -> dict[str, tuple[list[str], dict[str, Any]]]:
    """Edge members and the properties yeast_GEM sets on each hyperedge.

    hypernetx adds its own underscore-prefixed bookkeeping to the property dict in some
    versions (2.4.3 adds ``_level``; 2.4.0 does not), so those keys are dropped; every
    property the module writes is still compared exactly.
    """
    hg = gem.reaction_map
    return {
        eid: (
            list(hg.edges.elements[eid]),
            {
                key: value
                for key, value in dict(hg.edges[eid].properties).items()
                if not key.startswith("_")
            },
        )
        for eid in hg.edges
    }


def test_model_reads_the_sbml(gem: YeastGEM) -> None:
    """The SBML round trip keeps ids, equations, rules and metabolite fields."""
    rows = [
        (r.id, r.reaction, r.gene_reaction_rule, r.reversibility)
        for r in gem.model.reactions
    ]
    assert rows == [
        ("R1", "A_c <=> B_c", "(g1 and g2) or g3", True),
        ("R2", "2.0 B_c --> C_c", "g4", False),
        ("EX_A", "A_c <=> ", "", True),
    ]
    assert gem.model is gem.model
    assert gem.model_dir == os.path.join(gem.root, "yeast-GEM-9.0.2")


def test_gene_set(gem: YeastGEM) -> None:
    """The four GPR genes, from cobra's parsed ``model.genes``."""
    assert gem.gene_set == GeneSet(["g1", "g2", "g3", "g4"])


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("", [set()]),
        ("g4", [{"g4"}]),
        ("(g1 and g2) or g3", [{"g1", "g2"}, {"g3"}]),
        ("(a and b) or (c and d) or e", [{"a", "b"}, {"c", "d"}, {"e"}]),
    ],
)
def test_parse_gene_combinations(
    gem: YeastGEM, rule: str, expected: list[set[str]]
) -> None:
    """A disjunction of conjunctions splits into one gene set per OR term."""
    assert gem._parse_gene_combinations(rule) == expected


def test_parse_gene_combinations_ignores_nesting(gem: YeastGEM) -> None:
    """Finding: yeast_GEM.py:95 strips every parenthesis before splitting on ``or``.

    ``g1 and (g2 or g3)`` means {g1, g2} OR {g1, g3}, but with the parentheses removed
    it reads ``g1 and g2 or g3`` and parses to [{g1, g2}, {g3}], dropping g1 from the
    second unit.
    """
    assert gem._parse_gene_combinations("g1 and (g2 or g3)") == [{"g1", "g2"}, {"g3"}]


def test_reaction_map_edges(gem: YeastGEM) -> None:
    """Seven hyperedges: R1 x {g1,g2},{g3} x fwd/rev, R2 fwd, EX_A noGene fwd/rev."""
    assert _hyperedges(gem) == {
        "R1_comb0_fwd": (["A_c", "B_c"], {"genes": {"g1", "g2"}} | R1_FWD),
        "R1_comb0_rev": (["A_c", "B_c"], {"genes": {"g1", "g2"}} | R1_REV),
        "R1_comb1_fwd": (["A_c", "B_c"], {"genes": {"g3"}} | R1_FWD),
        "R1_comb1_rev": (["A_c", "B_c"], {"genes": {"g3"}} | R1_REV),
        "R2_comb0_fwd": (["B_c", "C_c"], R2_FWD),
        "EX_A_noGene_fwd": (["A_c"], EX_FWD),
        "EX_A_noGene_rev": (["A_c"], EX_REV),
    }


def test_reaction_map_with_induced_gene_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Induced {g1, g3}: R2 ({g4}) is dropped; {g1, g2} is kept with a printed warning."""
    _write_checkout(tmp_path)
    gem = YeastGEM(root=str(tmp_path), induced_gene_set=GeneSet(["g1", "g3"]))
    assert sorted(_hyperedges(gem)) == [
        "EX_A_noGene_fwd",
        "EX_A_noGene_rev",
        "R1_comb0_fwd",
        "R1_comb0_rev",
        "R1_comb1_fwd",
        "R1_comb1_rev",
    ]
    assert capsys.readouterr().out == (
        "Warning: Partial gene set overlap for edge R1_comb0. Genes not in set: "
        "{'g2'}. Full gene set for this reaction: "
        f"{ {'g1', 'g2'} }\n"
    )
    assert sorted(
        n
        for n, d in gem.bipartite_graph.nodes(data=True)
        if d["node_type"] == "reaction"
    ) == [
        "EX_A_noGene_fwd",
        "EX_A_noGene_rev",
        "R1_comb0_fwd",
        "R1_comb0_rev",
        "R1_comb1_fwd",
        "R1_comb1_rev",
    ]


def test_bipartite_edges_point_reaction_to_metabolite(gem: YeastGEM) -> None:
    """Exact edge list: reverse nodes swap reactant/product; stoichiometry is |coef|."""
    edges = [(u, v, d) for u, v, d in gem.bipartite_graph.edges(data=True)]

    def e(u: str, v: str, t: str, direction: str, s: float, r: str) -> Any:
        return (
            u,
            v,
            {
                "edge_type": t,
                "direction": direction,
                "stoichiometry": s,
                "reaction_id": r,
            },
        )

    assert edges == [
        e("R1_comb0_fwd", "A_c", "reactant", "forward", 1.0, "R1"),
        e("R1_comb0_fwd", "B_c", "product", "forward", 1.0, "R1"),
        e("R1_comb0_rev", "A_c", "product", "reverse", 1.0, "R1"),
        e("R1_comb0_rev", "B_c", "reactant", "reverse", 1.0, "R1"),
        e("R1_comb1_fwd", "A_c", "reactant", "forward", 1.0, "R1"),
        e("R1_comb1_fwd", "B_c", "product", "forward", 1.0, "R1"),
        e("R1_comb1_rev", "A_c", "product", "reverse", 1.0, "R1"),
        e("R1_comb1_rev", "B_c", "reactant", "reverse", 1.0, "R1"),
        e("R2_comb0_fwd", "B_c", "reactant", "forward", 2.0, "R2"),
        e("R2_comb0_fwd", "C_c", "product", "forward", 1.0, "R2"),
        e("EX_A_noGene_fwd", "A_c", "reactant", "forward", 1.0, "EX_A"),
        e("EX_A_noGene_rev", "A_c", "product", "reverse", 1.0, "EX_A"),
    ]
    assert gem.bipartite_graph is gem.bipartite_graph


def test_bipartite_node_attributes(gem: YeastGEM) -> None:
    """Metabolite nodes carry name/formula/compartment/charge; reactions their GPR."""
    nodes = dict(gem.bipartite_graph.nodes(data=True))
    assert nodes["B_c"] == {
        "node_type": "metabolite",
        "name": "beta",
        "composition": "C3H4O3",
        "compartment": "c",
        "charge": -1,
        "miriam": "",
        "inchi": "",
        "replacement_id": "",
    }
    assert nodes["R2_comb0_fwd"] == {
        "node_type": "reaction",
        "reaction_id": "R2",
        "direction": "forward",
        "genes": {"g4"},
        "equation": "2.0 B_c --> C_c",
        "reversibility": False,
        "reactants": ["B_c"],
        "products": ["C_c"],
        "subsystem": "",
    }
    assert nodes["EX_A_noGene_rev"] == {
        "node_type": "reaction",
        "reaction_id": "EX_A",
        "direction": "reverse",
        "genes": set(),
        "equation": "A_c <=> ",
        "reversibility": True,
        "reactants": [],
        "products": ["A_c"],
        "subsystem": "",
    }
    assert sorted(nodes) == [
        "A_c",
        "B_c",
        "C_c",
        "EX_A_noGene_fwd",
        "EX_A_noGene_rev",
        "R1_comb0_fwd",
        "R1_comb0_rev",
        "R1_comb1_fwd",
        "R1_comb1_rev",
        "R2_comb0_fwd",
    ]


def test_download_extracts_the_release_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing checkout is fetched from the tagged GitHub zip and extracted; zip removed."""
    staging = tmp_path / "staging"
    sbml = _write_checkout(staging)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.write(sbml, "yeast-GEM-9.0.2/model/yeast-GEM.xml")
    urls: list[str] = []

    def fake_get(url: str) -> SimpleNamespace:
        urls.append(url)
        return SimpleNamespace(content=buffer.getvalue(), raise_for_status=lambda: None)

    monkeypatch.setattr("torchcell.metabolism.yeast_GEM.requests.get", fake_get)
    root = tmp_path / "root"
    gem = YeastGEM(root=str(root))
    assert urls == [
        "https://github.com/SysBioChalmers/yeast-GEM/archive/refs/tags/v9.0.2.zip"
    ]
    assert os.listdir(root) == ["yeast-GEM-9.0.2"]
    assert [r.id for r in gem.model.reactions] == ["R1", "R2", "EX_A"]


def test_analyze_reactions_without_genes(gem: YeastGEM) -> None:
    """EX_A is the only gene-free reaction, and it is an exchange (one metabolite)."""
    assert analyze_reactions_without_genes(gem) == {
        "total_reactions": 3,
        "no_gene_reactions": ["EX_A"],
        "exchange_reactions": ["EX_A"],
        "transport_reactions": [],
        "other_reactions": [],
        "methods_consistent": True,
    }


def test_sanity_check_counts_no_bipartite_partitions(
    gem: YeastGEM, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: yeast_GEM.py:1064-1065 count nodes by a ``bipartite`` key never set.

    Nodes carry ``node_type``, so both partition counts print 0 and the "same type"
    check passes vacuously. The totals: 3 reactions, 3 metabolites, 4 genes, 7
    hyperedges, 3 hypergraph nodes, 12 bipartite edges, 10 bipartite nodes.
    """
    random.seed(0)
    sanity_check_metabolic_networks(gem, num_reactions=3)
    out = capsys.readouterr().out
    tail = out[out.index("===== Overall Statistics =====") :]
    assert tail == (
        "===== Overall Statistics =====\n"
        "Total reactions in model: 3\n"
        "Total metabolites in model: 3\n"
        "Total genes in model: 4\n"
        "Total edges in hypergraph: 7\n"
        "Total nodes in hypergraph: 3\n"
        "Total edges in bipartite graph: 12\n"
        "Total nodes in bipartite graph: 10\n"
        "Reaction nodes: 0\n"
        "Metabolite nodes: 0\n"
        "✓ All edges connect reactions to metabolites (bipartite property verified)\n"
        "✓ No isolated nodes found\n"
    )
    assert "Number of hyperedges for this reaction: 4\n" in out
    assert "Number of reaction nodes for this reaction: 1\n" in out


def test_bipartite_attribute_report(
    gem: YeastGEM, capsys: pytest.CaptureFixture[str]
) -> None:
    """The source-module ``test_bipartite_attributes`` helper prints R2's one node."""
    random.seed(0)
    yeast_gem_module.test_bipartite_attributes(gem, num_reactions=3)
    out = capsys.readouterr().out
    assert (
        "\nReaction Node: R2_comb0_fwd\n"
        "  Subsystem: \n"
        "  Direction: forward\n"
        "  Reversibility: False\n"
        "\n  Connected Metabolites:\n"
        "\n    Metabolite 1: B_c\n"
        "      Name: beta\n"
        "      Composition: C3H4O3\n"
        "      Compartment: c\n"
        "      Charge: -1\n"
        "      Edge Type: reactant\n"
        "      Stoichiometry: 2.0\n"
    ) in out


def test_plot_reaction_map_unknown_reaction(
    gem: YeastGEM, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No hyperedge for the id: prints and writes nothing."""
    out_path = tmp_path / "map.png"
    plot_reaction_map(gem, "NOPE", str(out_path))
    assert capsys.readouterr().out == "No edges found for reaction NOPE\n"
    assert not out_path.exists()


def test_plot_bipartite_network_counts_no_gene_nodes(
    gem: YeastGEM, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: yeast_GEM.py:729 selects ``node_type == "gene"``, which no node has.

    The graph's non-metabolite nodes are ``reaction`` nodes, so the "Genes" layer is
    always empty. For R2 the subgraph holds its two edges and two metabolites.
    """
    out_path = tmp_path / "bip.png"
    plot_bipartite_network(gem, reaction_id="R2", output_path=str(out_path))
    assert capsys.readouterr().out == (
        "\nNetwork Statistics:\n"
        "Number of genes: 0\n"
        "Number of metabolites: 2\n"
        "Number of edges: 2\n"
    )
    assert out_path.read_bytes()[:8] == PNG_MAGIC


def _capture_draw(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, dict[str, Any]]]:
    """Record every ``hnx.draw`` call made through the module's ``hnx`` import, then
    draw for real so the PNG is still written.
    """
    calls: list[tuple[Any, dict[str, Any]]] = []
    real_draw = hnx.draw

    def recording_draw(hypergraph: Any, **kwargs: Any) -> Any:
        calls.append((hypergraph, kwargs))
        return real_draw(hypergraph, **kwargs)

    monkeypatch.setattr("torchcell.metabolism.yeast_GEM.hnx.draw", recording_draw)
    return calls


def test_plot_reaction_map_writes_png(
    gem: YeastGEM,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1's four hyperedges, and only those, are drawn and the PNG is written."""
    calls = _capture_draw(monkeypatch)
    out_path = tmp_path / "map.png"
    plot_reaction_map(gem, "R1", str(out_path))
    assert capsys.readouterr().out == (
        f"Reaction map visualization saved to {out_path}\n"
    )
    assert len(calls) == 1
    assert sorted(calls[0][0].edges) == [
        "R1_comb0_fwd",
        "R1_comb0_rev",
        "R1_comb1_fwd",
        "R1_comb1_rev",
    ]
    assert out_path.read_bytes()[:8] == PNG_MAGIC


def test_plot_full_network_fails_on_hypernetx_sizes_kwarg(
    gem: YeastGEM, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding: yeast_GEM.py:605 passes ``sizes`` in ``nodes_kwargs``.

    Under hypernetx 2.4.0 that reaches ``EllipseCollection.set`` and raises, after the
    statistics are printed: 3 metabolites, 7 hyperedges, 6 of them on reversible
    reactions (all but R2).
    """
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "EllipseCollection.set() got an unexpected keyword argument 'sizes'"
        ),
    ):
        plot_full_network(gem, str(tmp_path / "full.png"))
    assert capsys.readouterr().out == (
        "\nNetwork Statistics:\n"
        "Number of metabolites (nodes): 3\n"
        "Number of reactions (edges): 7\n"
        "Number of reversible reactions: 6\n"
    )


def test_plot_random_network_spectral(
    gem: YeastGEM, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three sampled hyperedges drawn with the spectral layout.

    ``H.edges.elements`` lists the hyperedge ids sorted: EX_A_noGene_fwd,
    EX_A_noGene_rev, R1_comb0_fwd, R1_comb0_rev, R1_comb1_fwd, R1_comb1_rev,
    R2_comb0_fwd; after ``random.seed(0)``, ``random.sample`` of 3 draws R2_comb0_fwd,
    R1_comb0_rev, R1_comb1_rev. ``layout="spectral"`` passes networkx's ``spectral_layout`` with
    empty ``layout_kwargs``.
    """
    calls = _capture_draw(monkeypatch)
    random.seed(0)
    out_path = tmp_path / "random.png"
    plot_random_network(gem, n_edges=3, output_path=str(out_path), layout="spectral")
    assert len(calls) == 1
    hypergraph, kwargs = calls[0]
    assert list(hypergraph.edges) == ["R2_comb0_fwd", "R1_comb0_rev", "R1_comb1_rev"]
    assert kwargs["layout"] is spectral_layout
    assert kwargs["layout_kwargs"] == {}
    assert out_path.read_bytes()[:8] == PNG_MAGIC


def test_plot_full_bipartite_network_with_labels(
    gem: YeastGEM, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Whole graph: 0 "gene" nodes (see the Finding above), 3 metabolites, 12 edges."""
    out_path = tmp_path / "bip_all.png"
    plot_bipartite_network(gem, output_path=str(out_path), show_labels=True)
    assert capsys.readouterr().out == (
        "\nNetwork Statistics:\n"
        "Number of genes: 0\n"
        "Number of metabolites: 3\n"
        "Number of edges: 12\n"
    )
    assert out_path.read_bytes()[:8] == PNG_MAGIC
