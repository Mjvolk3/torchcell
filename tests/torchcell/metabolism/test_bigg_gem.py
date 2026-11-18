# tests/torchcell/metabolism/test_bigg_gem.py
# [[tests.torchcell.metabolism.test_bigg_gem]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/metabolism/test_bigg_gem.py
"""``BiggGEM`` over a two-reaction model deposited under ``tmp_path``.

Nothing here reaches the network: :func:`_deposited` writes the toy as BiGG-format
JSON and calls :meth:`BiggGEM.deposit`, which is the same path the provisioning
script uses. The toy carries an objective on ``BIOMASS`` so the growth detection
has something to find, and ``R1`` has a two-term GPR so the bipartite view has
two reaction nodes for it.

The behaviors pinned here are the ones a build depends on: the manifest records
the model's counts and its retrieval, the model file is re-hashed on every load,
a deposit that would change pinned bytes is refused, and the growth reaction is
read off the objective rather than off a subsystem string.
"""

import json
from pathlib import Path

import cobra
import pytest

from torchcell.metabolism.bigg_gem import (
    BiggGEM,
    GemManifest,
    ModelHashMismatchError,
    bigg_model_url,
)


def _toy_model() -> cobra.Model:
    model = cobra.Model("toy_bigg")
    a = cobra.Metabolite("a_c", name="A", formula="C3H4O3", compartment="c", charge=0)
    b = cobra.Metabolite("b_c", name="B", formula="C6H12O6", compartment="c", charge=-1)
    r1 = cobra.Reaction("R1", lower_bound=-1000.0, upper_bound=1000.0)
    r1.add_metabolites({a: -1.0, b: 1.0})
    r1.gene_reaction_rule = "(PP_0001 and PP_0002) or PP_0003"
    r1.subsystem = "central carbon"
    biomass = cobra.Reaction("BIOMASS", lower_bound=0.0, upper_bound=1000.0)
    biomass.add_metabolites({b: -1.0})
    biomass.subsystem = "Biomass and maintenance functions"
    model.add_reactions([r1, biomass])
    model.objective = "BIOMASS"
    return model


def _deposited(root: Path, model_id: str = "toy_bigg") -> BiggGEM:
    source = root / "source" / f"{model_id}.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    cobra.io.save_json_model(_toy_model(), str(source))
    gem = BiggGEM(model_id, gem_root=str(root / "gems"))
    gem.deposit(
        source_path=str(source),
        organism="Pseudomonas putida KT2440",
        source="test",
        source_url="https://example.invalid/toy_bigg.json",
        retrieval_method="direct_url",
        retrieval_command="direct_url('https://example.invalid/toy_bigg.json')",
        notes="toy",
    )
    return gem


def test_bigg_model_url_is_the_static_json_path() -> None:
    """The download URL is the model id under BiGG's static models directory."""
    assert (
        bigg_model_url("iJN1463") == "https://bigg.ucsd.edu/static/models/iJN1463.json"
    )


def test_deposit_writes_manifest_with_counts_and_retrieval(tmp_path: Path) -> None:
    """The manifest carries the model's own counts and how it was retrieved."""
    gem = _deposited(tmp_path)
    manifest = gem.manifest()
    assert isinstance(manifest, GemManifest)
    assert manifest.model_id == "toy_bigg"
    assert manifest.reaction_count == 2
    assert manifest.metabolite_count == 2
    assert manifest.gene_count == 3
    assert len(manifest.files) == 1
    record = manifest.files[0]
    assert record.path == "toy_bigg.json"
    assert record.role == "model"
    assert record.retrieval_method == "direct_url"
    assert record.sha256 == gem.summary()["sha256"]
    assert record.bytes == Path(gem.model_path).stat().st_size


def test_deposit_writes_sha256sums_matching_the_manifest(tmp_path: Path) -> None:
    """``SHA256SUMS`` lists the same digest the manifest pins."""
    gem = _deposited(tmp_path)
    line = (Path(gem.model_dir) / "SHA256SUMS").read_text().strip()
    digest, name = line.split("  ")
    assert name == "toy_bigg.json"
    assert digest == gem.manifest().files[0].sha256


def test_model_load_verifies_the_pinned_bytes(tmp_path: Path) -> None:
    """A model file edited after deposit raises instead of loading."""
    gem = _deposited(tmp_path)
    assert gem.model.id == "toy_bigg"
    payload = json.loads(Path(gem.model_path).read_text())
    payload["id"] = "tampered"
    Path(gem.model_path).write_text(json.dumps(payload))
    with pytest.raises(ModelHashMismatchError, match="hashes to"):
        BiggGEM("toy_bigg", gem_root=gem.gem_root).model


def test_deposit_refuses_to_overwrite_differing_bytes(tmp_path: Path) -> None:
    """Re-depositing different content for the same model id raises."""
    gem = _deposited(tmp_path)
    other = tmp_path / "other.json"
    model = _toy_model()
    model.reactions.get_by_id("R1").subsystem = "changed"
    cobra.io.save_json_model(model, str(other))
    with pytest.raises(ModelHashMismatchError, match="already holds different bytes"):
        gem.deposit(
            source_path=str(other),
            organism="Pseudomonas putida KT2440",
            source="test",
            source_url="https://example.invalid/other.json",
            retrieval_method="direct_url",
            retrieval_command="x",
        )


def test_deposit_is_idempotent_for_identical_bytes(tmp_path: Path) -> None:
    """Depositing the same bytes twice keeps one file and one digest."""
    gem = _deposited(tmp_path)
    first = gem.manifest().files[0].sha256
    again = _deposited(tmp_path)
    assert again.manifest().files[0].sha256 == first


def test_growth_reaction_comes_from_the_objective_not_the_subsystem(
    tmp_path: Path,
) -> None:
    """The toy's subsystem says nothing about growth; the objective names it."""
    gem = _deposited(tmp_path)
    assert gem.growth_reaction_ids == ("BIOMASS",)
    assert gem.model.reactions.get_by_id("BIOMASS").subsystem != "Growth"


def test_explicit_growth_reaction_ids_override_the_objective(tmp_path: Path) -> None:
    """A caller can name the growth reactions instead."""
    gem = BiggGEM(
        "toy_bigg", gem_root=str(tmp_path / "gems"), growth_reaction_ids=("R1",)
    )
    _deposited(tmp_path)
    assert gem.growth_reaction_ids == ("R1",)


def test_bipartite_graph_labels_growth_and_splits_the_gpr(tmp_path: Path) -> None:
    """The graph carries ``Growth`` on the biomass node and two nodes for R1."""
    gem = _deposited(tmp_path)
    graph = gem.bipartite_graph
    reactions = sorted(
        node for node, data in graph.nodes(data=True) if data["node_type"] == "reaction"
    )
    assert reactions == [
        "BIOMASS_noGene_fwd",
        "R1_comb0_fwd",
        "R1_comb0_rev",
        "R1_comb1_fwd",
        "R1_comb1_rev",
    ]
    assert graph.nodes["BIOMASS_noGene_fwd"]["subsystem"] == "Growth"
    assert (
        graph.nodes["BIOMASS_noGene_fwd"]["subsystem_source"]
        == "Biomass and maintenance functions"
    )
    assert graph.nodes["R1_comb0_fwd"]["genes"] == {"PP_0001", "PP_0002"}


def test_gene_set_is_every_gpr_gene(tmp_path: Path) -> None:
    """``gene_set`` is the model's own gene list."""
    gem = _deposited(tmp_path)
    assert set(gem.gene_set) == {"PP_0001", "PP_0002", "PP_0003"}


def test_reaction_ids_producing_names_only_producers(tmp_path: Path) -> None:
    """R1 produces b_c; BIOMASS consumes it."""
    gem = _deposited(tmp_path)
    assert gem.reaction_ids_producing("b_c") == ["R1"]


def test_summary_reports_the_counts_and_the_digest(tmp_path: Path) -> None:
    """``summary`` is what a build record stores."""
    gem = _deposited(tmp_path)
    summary = gem.summary()
    assert summary["reactions"] == 2
    assert summary["metabolites"] == 2
    assert summary["genes"] == 3
    assert summary["growth_reaction_ids"] == ["BIOMASS"]
    assert len(summary["sha256"]) == 64
